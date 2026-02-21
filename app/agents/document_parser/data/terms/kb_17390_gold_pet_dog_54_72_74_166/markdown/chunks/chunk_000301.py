from langchain_core.documents import Document

chunk = Document(
    page_content=('서 정한 상해를 말합니다.\n'
 '\uf000 제1항의 "외모특정상해"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의\n'
 '병원이나 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사(치과의사 제\n'
 '외) 면허를 가진 자에 의하여 내려져야 합니다. 또한 회사가 "외모특정상해"의\n'
 '조사나 확인을 위하여 필요하다고 인정하는 경우 검사결과, 진료기록부의 사본제\n'
 '출을 요청할 수 있습니다.-'),
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
