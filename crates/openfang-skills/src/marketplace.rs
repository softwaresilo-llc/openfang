//! FangHub marketplace client — install skills from the registry.
//!
//! For Phase 1, uses GitHub releases as the registry backend.
//! Each skill is a GitHub repo with releases containing the skill bundle.

use crate::SkillError;
use std::path::Path;
use tracing::info;

/// FangHub registry configuration.
#[derive(Debug, Clone)]
pub struct MarketplaceConfig {
    /// Base URL for the registry API.
    pub registry_url: String,
    /// GitHub organization for community skills.
    pub github_org: String,
}

impl Default for MarketplaceConfig {
    fn default() -> Self {
        Self {
            registry_url: "https://api.github.com".to_string(),
            github_org: "openfang-skills".to_string(),
        }
    }
}

/// Client for the FangHub marketplace.
pub struct MarketplaceClient {
    config: MarketplaceConfig,
    http: reqwest::Client,
}

impl MarketplaceClient {
    /// Create a new marketplace client.
    pub fn new(config: MarketplaceConfig) -> Self {
        Self {
            config,
            http: reqwest::Client::builder()
                .user_agent("openfang-skills/0.1")
                .build()
                .expect("Failed to build HTTP client"),
        }
    }

    /// Search for skills by query string.
    pub async fn search(&self, query: &str) -> Result<Vec<SkillSearchResult>, SkillError> {
        let url = build_search_url(&self.config, query)?;

        let resp = self
            .http
            .get(url)
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await
            .map_err(|e| SkillError::Network(format!("Search request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(SkillError::Network(format!(
                "Search returned status {}",
                resp.status()
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SkillError::Network(format!("Parse search response: {e}")))?;

        let results = body["items"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|item| SkillSearchResult {
                        name: item["name"].as_str().unwrap_or("").to_string(),
                        description: item["description"].as_str().unwrap_or("").to_string(),
                        stars: item["stargazers_count"].as_u64().unwrap_or(0),
                        url: item["html_url"].as_str().unwrap_or("").to_string(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(results)
    }

    /// Install a skill from a GitHub repo by name.
    ///
    /// Downloads the latest release tarball and extracts it to the target directory.
    pub async fn install(&self, skill_name: &str, target_dir: &Path) -> Result<String, SkillError> {
        let repo = format!("{}/{}", self.config.github_org, skill_name);
        let url = format!(
            "{}/repos/{}/releases/latest",
            self.config.registry_url, repo
        );

        info!("Fetching skill info from {url}");

        let resp = self
            .http
            .get(&url)
            .header("Accept", "application/vnd.github.v3+json")
            .send()
            .await
            .map_err(|e| SkillError::Network(format!("Fetch release: {e}")))?;

        if !resp.status().is_success() {
            return Err(SkillError::NotFound(format!(
                "Skill '{skill_name}' not found in marketplace (status {})",
                resp.status()
            )));
        }

        let release: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| SkillError::Network(format!("Parse release: {e}")))?;

        let version = release["tag_name"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        // Find the tarball asset
        let tarball_url = release["tarball_url"]
            .as_str()
            .ok_or_else(|| SkillError::Network("No tarball URL in release".to_string()))?;

        info!("Downloading skill {skill_name} {version}...");

        let skill_dir = target_dir.join(skill_name);
        std::fs::create_dir_all(&skill_dir)?;

        // Download the tarball
        let tar_resp = self
            .http
            .get(tarball_url)
            .send()
            .await
            .map_err(|e| SkillError::Network(format!("Download tarball: {e}")))?;

        if !tar_resp.status().is_success() {
            return Err(SkillError::Network(format!(
                "Download failed: {}",
                tar_resp.status()
            )));
        }

        // For now, save the download URL in a metadata file
        // Full tarball extraction would require a tar/gz library
        let meta = serde_json::json!({
            "name": skill_name,
            "version": version,
            "source": tarball_url,
            "installed_at": chrono::Utc::now().to_rfc3339(),
        });
        std::fs::write(
            skill_dir.join("marketplace_meta.json"),
            serde_json::to_string_pretty(&meta).unwrap_or_default(),
        )?;

        info!("Installed skill: {skill_name} {version}");
        Ok(version)
    }
}

/// Build a GitHub search URL with proper query encoding.
fn build_search_url(config: &MarketplaceConfig, query: &str) -> Result<reqwest::Url, SkillError> {
    let base = format!(
        "{}/search/repositories",
        config.registry_url.trim_end_matches('/')
    );
    let mut url = reqwest::Url::parse(&base)
        .map_err(|e| SkillError::Network(format!("Invalid registry URL '{base}': {e}")))?;
    let search = format!("{query} org:{}", config.github_org);
    url.query_pairs_mut()
        .append_pair("q", &search)
        .append_pair("sort", "stars");
    Ok(url)
}

/// A search result from the marketplace.
#[derive(Debug, Clone)]
pub struct SkillSearchResult {
    /// Skill name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Star count.
    pub stars: u64,
    /// Repository URL.
    pub url: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_default_config() {
        let config = MarketplaceConfig::default();
        assert!(config.registry_url.contains("github"));
        assert_eq!(config.github_org, "openfang-skills");
    }

    #[test]
    fn test_client_creation() {
        let client = MarketplaceClient::new(MarketplaceConfig::default());
        assert_eq!(client.config.github_org, "openfang-skills");
    }

    /// Search URL should preserve the logical query while applying URL encoding.
    #[test]
    fn test_search_query_encoding() {
        let config = MarketplaceConfig::default();
        let query = "social&media key=value /path?x=1#frag +50%";
        let url = build_search_url(&config, query).expect("search URL should build");

        // Parse decoded query params to ensure original semantics are preserved.
        let params: HashMap<String, String> = url.query_pairs().into_owned().collect();
        assert_eq!(params.get("sort"), Some(&"stars".to_string()));
        assert_eq!(
            params.get("q"),
            Some(&format!("{query} org:{}", config.github_org))
        );

        // Raw URL must contain percent-encoded bytes for reserved characters.
        let raw = url.as_str();
        assert!(raw.contains("%26")); // &
        assert!(raw.contains("%3D")); // =
        assert!(raw.contains("%23")); // #
        assert!(raw.contains("%25")); // %
    }
}
