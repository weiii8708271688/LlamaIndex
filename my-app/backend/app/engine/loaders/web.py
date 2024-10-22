from pydantic import BaseModel, Field


class CrawlUrl(BaseModel):
    base_url: str
    prefix: str
    max_depth: int = Field(default=1, ge=0)


class WebLoaderConfig(BaseModel):
    driver_arguments: list[str] = Field(default=None)
    urls: list[CrawlUrl]


def get_web_documents(config: WebLoaderConfig):
    from llama_index.readers.web import WholeSiteReader
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    service = Service('/home/erichuang/llama/my-app2/backend/app/engine/loaders/chromedriver')
    
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--headless')  # 如果在無頭環境中運行
    options.add_argument('--disable-gpu')  # 在某些系統上可能需要
    options.add_argument('--remote-debugging-port=9222')  # 添加調試端口
    driver_arguments = config.driver_arguments or []
    for arg in driver_arguments:
        options.add_argument(arg)

    docs = []
    for url in config.urls:
        scraper = WholeSiteReader(
            prefix=url.prefix,
            max_depth=url.max_depth,
            driver=webdriver.Chrome(service=service, options=options),
        )
        docs.extend(scraper.load_data(url.base_url))

    return docs
