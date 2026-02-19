from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[보험가입금액 제한]\n'
 '피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다.\n'
 '[일부보장 제외]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 특정 질병 또는 특정 신 체 부위를 보장에서 제외하는 방법을 '
 '말합니다.\n'
 '[보험금 삭감]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 51},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000212',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
