from langchain_core.documents import Document

chunk = Document(
    page_content=('- 표Ⅱ(치아파절제외))에 정한 상병을 말합니다.\n'
 '- \uf000 "골절진단(치아파절제외)"의 진단은 의료법 제3조(의료기관)에서 정한 국내의 병 상\n'
 '- 원이나 의원 또는 국외의 의료관련법에서 정한 의료기관의 의사 면허를 가진 자 해\n'
 '- 에 의하여 내려져야 합니다. 또한 회사가 "골절진단(치아파절제외)"의 조사나 확 및\n'
 '- 인을 위하여 필요하다고 인정하는 경우 검사결과, 진료기록부의 사본제출을 요청 질\n'
 '- 병\n'
 '- 할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000332',
              'chunk_char_len': 237,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
