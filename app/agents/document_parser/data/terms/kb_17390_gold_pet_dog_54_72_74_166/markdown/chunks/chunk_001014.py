from langchain_core.documents import Document

chunk = Document(
    page_content=('| 천식, 천식지속 상태 | J45, J46 |\n'
 '| 폐렴 | J12~J18 |\n'
 '| [B01.2+: 수두폐렴(J17.1*)] | B01.2+ |\n'
 '| [B05.2+: 폐렴이 합병된 홍역(J17.1*)] | B05.2+ |\n'
 '| [B25.0+: 거대세포바이러스폐렴(J17.1*)] | B25.0+ |\n'
 '| [B58.3+: 폐 톡소포자충증(J17.3*)] | B58.3+ |\n'
 '| 재향군인병 | A48.1 |'),
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
