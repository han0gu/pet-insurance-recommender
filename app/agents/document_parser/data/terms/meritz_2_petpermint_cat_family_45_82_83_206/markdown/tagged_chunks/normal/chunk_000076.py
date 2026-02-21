from langchain_core.documents import Document

chunk = Document(
    page_content=('가 청약서에 자필서명을 하지 않은 때에는 계약자는 계약이\n'
 '성립한 날부터 3개월 이내에 계약을 취소할 수 있습니다.# 【자필서명】계약자가 성명기입란에 본인의 성명을 기재하고, 날인란\n'
 '에 사인(signature) 또는 도장을 찍는 것을 말합니다.\n'
 '전자서명법 제2조 제2호에 따른 전자서명을 포함합니다.# 【 전자서명법 제2조 제2호에 따른 전자서명 】"전자서명"이란 다음 각 목의 '
 '사항을 나타내는 데 이용하\n'
 '기 위하여 전자문서에 첨부되거나 논리적으로 결합된 전\n'
 '자적 형태의 정보를 말한다.가. 서명자의 신원'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000076',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
