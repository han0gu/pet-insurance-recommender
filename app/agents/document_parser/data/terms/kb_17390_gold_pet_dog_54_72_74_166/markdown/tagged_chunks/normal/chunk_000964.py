from langchain_core.documents import Document

chunk = Document(
    page_content=('| 요추 및 골반의 골절 | S32 표 |\n'
 '| 어깨 및 위팔의 골절 | S42 |\n'
 '| 아래팔의 골절 손목 및 손부위의 골절 | S52 |\n'
 '| 대퇴골의 골절 | S62 S72 |\n'
 '| 발목을 포함한 아래다리의 골절 | S82 법 |\n'
 '| 발목을 제외한 발의 골절 | S92 ㆍ |\n'
 '| 여러 신체부위를 침범한 골절 | T02 규정 |\n'
 '| 척추의 상세불명 부위의 골절 | T08 |\n'
 '| 팔의 상세불명 부위의 골절 | T10 |\n'
 '| 다리의 상세불명 부위의 골절 | T12 |\n'
 '| 상세불명의 신체부위의 골절 | T14.2 |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000964',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
