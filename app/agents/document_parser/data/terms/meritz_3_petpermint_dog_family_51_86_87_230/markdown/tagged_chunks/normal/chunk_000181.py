from langchain_core.documents import Document

chunk = Document(
    page_content=('의무사항이 중요한 사항에 해당되는 사유를「반대증거가 있\n'
 '는 경우 이의를 제기할 수 있습니다」라는 문구와 함께 계\n'
 '약자에게 서면 또는 전자문서 등으로 알려 드립니다. 회사\n'
 '가 전자문서로 안내하고자 할 경우에는 계약자에게 서면 또\n'
 '는 「전자서명법」 제2조 제2호에 따른 전자서명으로 동의\n'
 '를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니\n'
 '다. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자\n'
 '문서는 송신되지 않은 것으로 봅니다. 회사는 전자문서가\n'
 '수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으\n'
 '로 다시 알려드립니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000181',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
