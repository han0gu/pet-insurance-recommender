from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 실시되어져야 하며, 자격을 갖춘 임상심리전문\n'
 '- 가가 시행하고 작성하여야 한다.\n'
 '- 차) 정신행동장해 진단 전문의는 정신건강의학과 전문\n'
 '- 의를 말한다.\n'
 '- 카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있\n'
 '- 는 뇌자기공명촬영, 뇌전산화촬영, 뇌파 등 객관적\n'
 '- 근거를 기초로 평가한다. 다만, 보호자나 환자의\n'
 '- 진술, 감정의의 추정 혹은 인정, 한국표준화가 이\n'
 '- 루어지지 않고 신빙성이 적은 검사들(뇌 SPECT 등)\n'
 '- 은 객관적 근거로 인정하지 않는다.\n'
 '- 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000616',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
