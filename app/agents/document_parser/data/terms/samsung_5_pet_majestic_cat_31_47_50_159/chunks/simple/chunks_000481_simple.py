from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 특별약관에서「특정법정감염병」이라 함은 [별표-질병관련1]특정법정감염병 분류 표에서 정한 질병을 말합니다. ② 특정법정감염병의 '
 '진단확정은 감염병병원체 확인기관([별표-질병관련1.1]감염병병원 체 확인기관 참조)에서 감염병환자로 확진된 경우를 말하며, 감염병의 예방 '
 '및 관리에 관한 법률 시행규칙상 감염병환자등의 진단 기준에 따른 감염병환자, 의사환자를 포 함하고, 병원체보유자는 해당되지 않습니다'),
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
 'indexing': {'chunk_id': 'chunk_000481',
              'chunk_char_len': 224,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
