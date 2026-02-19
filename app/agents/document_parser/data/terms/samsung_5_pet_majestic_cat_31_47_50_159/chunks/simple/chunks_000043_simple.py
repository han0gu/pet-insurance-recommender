from langchain_core.documents import Document

chunk = Document(
    page_content=('100만원인 경우\n'
 '→ 총 중도인출 가능액 = 100만원 × 80% = 80만원 → 기 신청한 보험계약대출금이 있는 경우(원금과 이자의 합계를 30만원으로 '
 '가정) 중도인출 가능액 ＝ 80만원(총 중도인출 가능액) － 30만원 ＝ 50만원\n'
 '제11조 (만기환급금의 지급)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000043',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
