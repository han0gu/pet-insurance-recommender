from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.\n'
 '① 보험금 청구서(회사양식) ② 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정 부기관발행 신분증, 본인이 아닌 경우에는 본인의 인 '
 '감증명서, 본인서명사실확인서 또는 안전성과 신뢰성 이 확보된 전자적 수단을 활용한 보험수익자 의사표시 의 확인방법 포함) ③ 손해배상금 '
 '및 그 밖의 비용을 지급하였음을 명하는 서류 ④ 회사가 요구하는 그 밖의 서류\n'
 '제5조(보험금의 지급절차)'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000591',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
