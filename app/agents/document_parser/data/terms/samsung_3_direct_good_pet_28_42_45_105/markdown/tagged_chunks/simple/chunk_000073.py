from langchain_core.documents import Document

chunk = Document(
    page_content=('- 급사유로 한 경우. 다만, 심신박약자가 계약을 체결하거나 소속 단체의 규약에 따\n'
 '- 라 단체보험의 피보험자가 될 때에 의사능력이 있는 경우 계약이 유효합니다.\n'
 '<용어풀이># [심신상실자(心神喪失者)]의식은 있으나 장애의 정도가 심하여 자신의 행위 결과를 합리적으로 판단할 능력을 갖지 못한\n'
 '사람은 마해 1# 의사를 결정할 능력이 미약한 사람을 말합니다.3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 '
 '초과되었을 경\n'
 '우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 계약나이에 도달한 경우에는'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000073',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
