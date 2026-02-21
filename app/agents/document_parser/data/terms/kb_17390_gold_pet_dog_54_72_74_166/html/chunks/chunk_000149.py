from langchain_core.documents import Document

chunk = Document(
    page_content=("철회)</p><br><h1 id='179' style='font-size:14px'>\uf000 계약자는</h1><br><p "
 "id='180' data-category='paragraph' style='font-size:14px'>보험증권을 받은 날부터 15일 "
 '이내에 그 청약을 철회할 수 있습니다'),
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
