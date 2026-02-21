from langchain_core.documents import Document

chunk = Document(
    page_content=('자격을 갖춘 임상심리전문가가 시행하고 작성하여야 한다.\n'
 '차) 정신행동장해 진단 전문의는 정신건강의학과 전문의를 말한다.\n'
 '카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있는 뇌자기공명촬영, 뇌전\n'
 '산화촬영, 뇌파 등 객관적 근거를 기초로 평가한다. 다만, 보호자나 환자의\n'
 '진술, 감정의의 추정 혹은 인정, 한국표준화가 이루어지지 않고 신빙성이\n'
 '적은 검사들(뇌 SPECT 등)은 객관적 근거로 인정하지 않는다.\n'
 '타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000840',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
