from langchain_core.documents import Document

chunk = Document(
    page_content=("증가된 경우에는 통지를 받은 날부터 1개월 이내에 보험료의 증액을 청구하거</h1><br><h1 id='14' "
 "style='font-size:14px'>나 특별약관을 해지할 수 있습니다.</h1><br><p id='15' "
 "data-category='paragraph' style='font-size:14px'>제18조(알릴 의무 위반의 "
 "효과)<br>\uf000 회사는 아래와 같은 사실이 있을 경우에는 손해의 발생여부에 관계없이 그 사실</p><br><p id='16' "
 "data-category='list'"),
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
