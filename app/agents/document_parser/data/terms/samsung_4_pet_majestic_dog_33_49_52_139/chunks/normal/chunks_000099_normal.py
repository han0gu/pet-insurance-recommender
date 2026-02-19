from langchain_core.documents import Document

chunk = Document(
    page_content=('으로 약관의 중요한 내용을 설명할 수 있습니다.\n'
 '1. 전화를 이용하여 청약내용, 보험료납입, 보험기간, 계약 전 알릴 의무, 약관의 중요 한 내용 등 계약을 체결하는 데 필요한 사항을 '
 '질문 또는 설명하는 방법. 이 경우 계약자의 답변과 확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것 으로 봅니다.\n'
 '<용어풀이>\n'
 '[약관의 중요한 내용]\n'
 '금융소비자 보호에 관한 법률 제19조(설명의무) 등에서 정한 다음의 내용을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 42},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
