from langchain_core.documents import Document

chunk = Document(
    page_content=(". 보험료자동납입</p><br><p id='1' data-category='paragraph' "
 "style='font-size:14px'>제1조(보험료의 납입)<br>계약자는 제2회 이후의 보험료부터 이 특별약관에 따라 계약자의 "
 "거래은행 지정계좌</p><br><h1 id='2' style='font-size:14px'>를 이용하여 보험료를 자동 "
 "납입합니다.</h1><p id='3' data-category='paragraph' "
 "style='font-size:14px'>제2조(보험료의 영수)<br>자동납입일자는 이 청약서에 기재된"),
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
 'indexing': {'chunk_id': 'chunk_001359',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
