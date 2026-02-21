from langchain_core.documents import Document

chunk = Document(
    page_content=("특별약관은 계약자와 회사 사이에 보험증권에 기재된 반려동물의 상해 또는 질병</p><br><p id='105' "
 "data-category='paragraph' style='font-size:14px'>으로 인한 위험을 보장하기 위하여 "
 "체결됩니다.</p><p id='106' data-category='paragraph' "
 "style='font-size:14px'>제2조(용어의 정의)<br>이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의 다른 조항에서 "
 "달리 정의</p><br><h1 id='107'"),
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
