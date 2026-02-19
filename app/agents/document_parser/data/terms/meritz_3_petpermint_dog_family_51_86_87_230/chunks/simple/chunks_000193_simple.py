from langchain_core.documents import Document

chunk = Document(
    page_content=('기타 지불 증빙서류 등)\n'
 '③ 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정 부기관발행 신분증, 본인이 아닌 경우에는 본인의 인 감증명서, 본인서명사실확인서 '
 '또는 안전성과 신뢰성 이 확보된 전자적 수단을 활용한 보험수익자 의사표시 의 확인방법 포함) ④ 기타 보험수익자가 보험금의 수령에 '
 '필요하여 제출하 는 서류\n'
 '\uf000 제1항 제2호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국내의 동물병원에서 수의사에 의해 발급한 '
 '것이어야 합니다.\n'
 '【수의사법 제12조(진단서 등)】'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 92},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000193',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
