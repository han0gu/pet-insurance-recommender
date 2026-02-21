from langchain_core.documents import Document

chunk = Document(
    page_content=('. "이물제거(내시경)"이란 반려동물의 위장 등 내부의 이물질을 제거하기 위하<br>여 수술을 동반하지 않고 내시경 및 내시경포셉을 '
 '이용하여 비침습적으로 시<br>행하는 의료행위를 말합니다.<br>2'),
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
