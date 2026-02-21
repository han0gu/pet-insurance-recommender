from langchain_core.documents import Document

chunk = Document(
    page_content=('및 각각의 보험료<br>별<br>청약의 철회에 관한 사항(기한ㆍ행사방법ㆍ효과 등)<br>표<br>지급한도, 면책사항, 감액지급 사항 등 '
 '보험금 지급제한 조건<br>고지의무 및 통지의무 위반의 효과<br>계약의 취소 및 무효에 관한 사항<br>해약환급금에 관한 사항(납부한 '
 '보험료보다 적거나 없을 수 있다는 사실 포함)<br>민원처리 및 분쟁조정절차에 관한 사항 법<br>만기시 자동갱신되는 보험계약의 경우 '
 '자동갱신의 조건 ㆍ<br>저축성 보험계약의 공시이율 규정<br>유배당 보험계약의 경우 계약자 배당에 관한 사항<br>그 밖에'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
