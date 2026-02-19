from langchain_core.documents import Document

chunk = Document(
    page_content=('제2관 개별사항\n'
 '제1조 (보험금의 지급사유)\n'
 '회사는 피보험자가 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간」이라 합 니다) 중에 「특정법정감염병」으로 감염병의 예방 및 '
 '관리에 관한 법률 제11조(의사 등 의 신고)에 따라 신고되어 특정법정감염병 환자로 진단 확정되었을 때에는 보험증권에 기 재된 이 '
 '특별약관의 보험가입금액을 특정법정감염병 진단비로 보험수익자에게 지급합니 다.\n'
 '제 2조 (보험금 지급에 관한 세부규정)'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 88},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000476',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
