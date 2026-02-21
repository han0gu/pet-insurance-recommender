from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하며, 자격을 갖춘 임상심리전문가가 시행하고 작성하여야 한다.\n'
 '- 차) 정신행동장해 진단 전문의는 정신건강의학과 전문의를 말한다.\n'
 '- 카) 정신행동장해는 뇌의 기능 및 결손을 입증할 수 있는 뇌자기공명촬\n'
 '- 영, 뇌전산화촬영, 뇌파 등 객관적 근거를 기초로 평가한다. 다만,\n'
 '- 보호자나 환자의 진술, 감정의의 추정 혹은 인정, 한국표준화가 이\n'
 '- 루어지지 않고 신빙성이 적은 검사들(뇌 SPECT 등)은 객관적 근거로\n'
 '- 인정하지 않는다.\n'
 '- 타) 각종 기질성 정신장해와 외상후 뇌전증에 한하여 보상한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition',
            'risk_domains': ['digestive', 'head', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000939',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
