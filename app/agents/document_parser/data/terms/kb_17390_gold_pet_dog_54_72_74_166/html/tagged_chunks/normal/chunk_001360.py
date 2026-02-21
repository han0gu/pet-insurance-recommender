from langchain_core.documents import Document

chunk = Document(
    page_content=("영수)<br>자동납입일자는 이 청약서에 기재된 보험료 납입해당일에도 불구하고 매월 회사가 정</p><br><p id='4' "
 "data-category='paragraph' style='font-size:14px'>하는 날 중 계약자가 희망하는 일자로 "
 "합니다.</p><p id='5' data-category='paragraph' style='font-size:14px'>제3조(계약 후 "
 "알릴 의무)<br>계약자는 지정계좌의 번호가 변경 또는 거래 정지된 경우에는 이 사실을 즉시 회사에</p><br><h1 id='6'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_001360',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
