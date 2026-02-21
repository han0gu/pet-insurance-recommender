from langchain_core.documents import Document

chunk = Document(
    page_content=('- 차) 정신행동장해 진단 전문의는 정신건강의학과 전문\n'
 '- 의를 말한다.\n'
 '- 카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있\n'
 '- 는 뇌자기공명촬영, 뇌전산화촬영, 뇌파 등 객관적\n'
 '- 근거를 기초로 평가한다. 다만, 보호자나 환자의\n'
 '- 진술, 감정의의 추정 혹은 인정, 한국표준화가 이\n'
 '- 루어지지 않고 신빙성이 적은 검사들(뇌 SPECT 등)\n'
 '- 은 객관적 근거로 인정하지 않는다.\n'
 '- 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여\n'
 '- 보상한다.\n'
 '- 파) 외상후 스트레스장애, 우울증(반응성) 등의 질환,'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000690',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
