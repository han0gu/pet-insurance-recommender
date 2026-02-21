from langchain_core.documents import Document

chunk = Document(
    page_content=('- 방법을 진단의 기초로 할 수 없는 경우에 한하여 피보험자가 특정법정감염병으로 진\n'
 '- 단 또는 치료를 받고 있었음을 증명할 수 있는 문서화된 기록 또는 증거를 진단확정\n'
 '- 의 기초로 할 수 있습니다.\n'
 '# <관련법규># [감염병의 예방 및 관리에 관한 법률 시행규칙 [별표2] 감염병환자등의 진단기준]∙ 감염병환자 : 해당 감염병에 '
 '부합되는 임상적 특징을 나타내면서 특정 검사방법으로 감염병 병원\n'
 '체가 확인된 사람\n'
 '∙ 의사환자 : 임상적 특징 및 역학적 연관성을 고려할 때 해당 감염병이 의심되나 감염병 병원체가\n'
 '확인되지 않은 사람'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000407',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
