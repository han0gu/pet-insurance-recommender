from langchain_core.documents import Document

chunk = Document(
    page_content=('| <table><thead></thead><tbody><tr><td></td></tr><tr><td>예 시 3세인 피보험자 반려동물이 '
 '3년만기로 갱신하는 경우 아래 예시에서 최초 계약시 납입할 특약보험료는 가입시점의 3세 보험료인 특 5,000원입니다. 이후 첫 번째 '
 '갱신계약의 특약보험료는 가입 3년후 새롭게 산출 별 한 보험료표를 적용받는데, 우선 피보험자 반려동물의 나이증가(3세 → 6세)로 약 '
 '인한 보험료의 증가분과 새롭게 산출된 보험료의 인하분이 함께 반영되어 6,200 관 원을 납입합니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
