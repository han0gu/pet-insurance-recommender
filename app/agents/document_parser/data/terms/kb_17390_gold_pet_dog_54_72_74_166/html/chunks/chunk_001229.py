from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>\uf000 계약자, 피보험자 또는 이들의 대리인의 사기에 의하여 계약이 성립되었음을 "
 '회<br>사가 증명하는 경우에는 계약일부터 5년 이내(사기사실을 안 날부터 1개월 이내)<br>에 계약을 취소할 수 '
 '있습니다.<br>\uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 계약자에게<br>돌려드립니다.</p><h1 '
 "id='27' style='font-size:16px'>\uf000</h1><br><h1 id='28' "
 "style='font-size:16px'>제21조(조사)</h1><br><p"),
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
