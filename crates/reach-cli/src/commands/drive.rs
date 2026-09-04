pub use reach_cli::drive::DriveArgs;

pub async fn run(args: DriveArgs) -> anyhow::Result<()> {
    reach_cli::drive::run(args).await
}
