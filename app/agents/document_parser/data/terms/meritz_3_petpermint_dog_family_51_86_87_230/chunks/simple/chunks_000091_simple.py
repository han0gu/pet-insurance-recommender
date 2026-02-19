from langchain_core.documents import Document

chunk = Document(
    page_content=('약 전 알릴 의무, 약관의 중요한 내용 등 계약을 체결 하는 데 필요한 사항을 질문 또는 설명하는 방법. 이 경우 계약자의 답변과 '
 '확인내용을 음성 녹음함으로써 약관의 중요한 내용을 설명한 것으로 봅니다.\n'
 '【통신판매계약】\n'
 '전화·우편·인터넷 등 통신수단을 이용하여 체결하는 계약을 말합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 70},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
