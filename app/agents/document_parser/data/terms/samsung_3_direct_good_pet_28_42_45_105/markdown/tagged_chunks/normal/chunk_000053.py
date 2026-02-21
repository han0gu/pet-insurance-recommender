from langchain_core.documents import Document

chunk = Document(
    page_content=('- ② 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보\n'
 '- 험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있\n'
 '- 습니다.\n'
 '<용어풀이>[보험가입금액 제한]\n'
 '피보험자가 가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다.[일부보장 제외]\n'
 '일반적인 경우보다 위험이 높은 피보험자가 가입하기 위한 방법의 하나로, 특정 질병 또는 특정 신'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000053',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
