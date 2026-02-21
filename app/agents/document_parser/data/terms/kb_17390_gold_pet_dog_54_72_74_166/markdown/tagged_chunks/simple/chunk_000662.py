from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2. | 지급사유 및 보상 관련 용어 |\n'
 '| --- | --- |\n'
 '| 용 어 | 정 의 |\n'
 '| --- | --- |\n'
 '| 중요한 사항 | 계약전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 보 '
 '장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하 는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합니다. |\n'
 '| 자기부담금 | 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보 험자가 부담하는 일정 금액을 말합니다. |'),
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
 'indexing': {'chunk_id': 'chunk_000662',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
