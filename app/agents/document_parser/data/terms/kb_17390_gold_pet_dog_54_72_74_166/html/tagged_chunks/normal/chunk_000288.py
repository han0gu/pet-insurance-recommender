from langchain_core.documents import Document

chunk = Document(
    page_content=('. 예시) 보험계약일이 8월 15일인 경우 보험년도 기준 매1년은 해당년도 8월 15 일부터 다음 해 8월 '
 "14일까지입니다.</td></tr></tbody></table><br><table id='114' "
 "style='font-size:16px'><thead></thead><tbody><tr><td></td></tr><tr><td>예 시 "
 '중도인출금의 한도 중도인출 시점에 "보험료 및 해약환급금 산출방법서"에 의해 산출된 기본계약 공 해약환급금과 적립부분 해약환급금 중 적은 '
 '금액이 100만원인 경우 통 ⇒ 총 중도인출 가능액 ='),
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
 'indexing': {'chunk_id': 'chunk_000288',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
