from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 수의사는 직접 진료하거나 검안한 동물에 대한 진단 서, 검안서, 증명서 또는 처방전의 발급을 요구받았 을 때에는 정당한 사유 없이 '
 '이를 거부하여서는 아니 된다. ④ 제1항부터 제3항까지의 규정에 따른 진단서, 검안서, 증명서 또는 처방전의 서식, 기재사항, 그 밖에 '
 '필요 한 사항은 농림축산식품부령으로 정한다. ⑤ 제1항에도 불구하고 농림축산식품부장관에게 신고한 축산농장에 상시고용된 수의사는 해당 '
 '농장의 가축에 게 투여할 목적으로 동물용 의약품에 대한 처방전을 발급할 수 있다'),
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
 'indexing': {'chunk_id': 'chunk_000195',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
