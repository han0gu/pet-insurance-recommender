from langchain_core.documents import Document

chunk = Document(
    page_content=('| 질병 | 상해를 제외한 상병을 모두 포함합니다. |\n'
 '| 중요한 사항 | 계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 보 '
 '장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하 는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합니다. |\n'
 '| 보험가입금액 | 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가 지급할 최대 보험금을 말합니다. |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000447',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
