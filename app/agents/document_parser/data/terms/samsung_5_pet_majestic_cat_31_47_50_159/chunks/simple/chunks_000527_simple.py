from langchain_core.documents import Document

chunk = Document(
    page_content=('2. 질병: 상해를 제외한 상병을 모두 포함합니다. 3. 보험가입금액 : 회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가 '
 '지급할 최대 보험금을 말합니다. 4. 자기부담금 : 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부 담하는 일정 금액을 '
 '말합니다. 5. 보험금 분담 : 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약( 공제계약을 포함합니다)이 있을 경우 '
 '비율에 따라 손해를 보상합니다.\n'
 '<용어풀이>\n'
 '[공제계약]'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000527',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
