from langchain_core.documents import Document

chunk = Document(
    page_content=('자) 심리학적 평가보고서는 정신건강의학과 의료기관에 서 실시되어져야 하며, 자격을 갖춘 임상심리전문 가가 시행하고 작성하여야 한다. 차) '
 '정신행동장해 진단 전문의는 정신건강의학과 전문 의를 말한다. 카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있 는 뇌자기공명촬영, '
 '뇌전산화촬영, 뇌파 등 객관적 근거를 기초로 평가한다. 다만, 보호자나 환자의 진술, 감정의의 추정 혹은 인정, 한국표준화가 이 루어지지 '
 '않고 신빙성이 적은 검사들(뇌 SPECT 등) 은 객관적 근거로 인정하지 않는다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 228},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000819',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
