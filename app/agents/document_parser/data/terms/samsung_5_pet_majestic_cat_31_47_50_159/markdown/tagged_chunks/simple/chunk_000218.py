from langchain_core.documents import Document

chunk = Document(
    page_content=('② 제1항과 관련하여 통신판매계약의 경우, 회사는 계약자가 가입한 특별약관만 포함한\n'
 '약관을 드리며, 전화를 이용하여 체결하는 계약은 계약자의 동의를 얻어 다음의 방법\n'
 '으로 약관의 중요한 내용을 설명할 수 있습니다.1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요\n'
 '한 내용 등 계약을 체결하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우\n'
 '계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것'),
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
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
