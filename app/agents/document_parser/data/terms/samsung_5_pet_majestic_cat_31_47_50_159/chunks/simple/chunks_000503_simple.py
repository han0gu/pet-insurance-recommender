from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 개별사항\n'
 '제1조 (보험금의 지급사유)\n'
 '① 회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중에 상해 또는 진단확정된 질병으로 아래에 '
 '정한 창상봉합술을 받은 경우 각 보 장별 1일 1회, 총 연간 3회에 한하여 아래에 기재된 지급금액을 창상봉합술 치료비(1일1 회한) '
 '(이하 「창상봉합술 치료비」 라 합니다)로 보험수익자에게 지급합니다.\n'
 '보장구분 | 보험금 지급사유 | 지급금액'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 93},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000503',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
