from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자, 피보험자<br>5. 보험가입금액(배상책임의 경우 보상한도액) 등 기타 계약의 내용<br>회사는 계약자가 제1회 보험료를 '
 "납입한 때부터 1년 이상 지난 유효한 계약으로</p><br><p id='32' data-category='paragraph' "
 "style='font-size:16px'>\uf000</p><br><p id='33' data-category='list' "
 "style='font-size:14px'>서 그 보험종목의 변경을 요청할 때에는 회사의 사업방법서에서 정하는 방법에<br>따라 이를 "
 '변경하여 드립니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000862',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
