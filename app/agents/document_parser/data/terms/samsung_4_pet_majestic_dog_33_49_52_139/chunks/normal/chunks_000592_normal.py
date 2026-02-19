from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 전자문서로 안 내하고자 할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 전 자서명으로 동의를 얻어 '
 '수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자 의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 '
 '봅니다. 회사는 전자문서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립니다. ⑤ 제1항 제2호에 의한 계약의 '
 '해지가 보험금 지급사유 발생 후에 이루어진 경우에는 제 제12조(계약 후 알릴 의무) 제4항 또는 제5항에 따라 보험금을 지급합니다'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000592',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
