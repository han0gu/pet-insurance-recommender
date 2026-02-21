from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 보험가입금액 : 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가\n'
 '- 지급할 최대 보험금을 말합니다.\n'
 '- 4. 자기부담금 : 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부\n'
 '- 담하는 일정 금액을 말합니다.\n'
 '- 5. 보험금 분담 : 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(\n'
 '- 공제계약을 포함합니다)이 있을 경우 비율에 따라 손해를 보상합니다.\n'
 '<용어풀이># [공제계약]유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000448',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
