from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>일반금융소비자</h1><br><h1 id='189' "
 "style='font-size:14px'>전문금융소비자가 아닌 계약자를 말합니다.</h1><p id='190' "
 "data-category='paragraph' style='font-size:14px'>관 련 법 규 금융소비자보호에 관한 "
 '법률<br>제46조(청약의 철회) ① 금융상품판매업자등과 대통령령으로 각각 정하는 보장<br>성 상품, 투자성 상품, 대출성 상품 또는 '
 '금융상품자문에 관한 계약의 청약을 한<br>일반금융소비자는 다음 각 호의'),
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
