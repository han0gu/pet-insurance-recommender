from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[감염병의 예방 및 관리에 관한 법률 시행규칙 [별표2] 감염병환자등의 진단기준]\n'
 '∙ 감염병환자 : 해당 감염병에 부합되는 임상적 특징을 나타내면서 특정 검사방법으로 감염병 병원 체가 확인된 사람 ∙ 의사환자 : 임상적 '
 '특징 및 역학적 연관성을 고려할 때 해당 감염병이 의심되나 감염병 병원체가 확인되지 않은 사람 ∙ 병원체보유자 : 임상증상을 나타내지 '
 '않으나 감염병병원체가 확인된 사람'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
