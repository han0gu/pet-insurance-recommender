from langchain_core.documents import Document

chunk = Document(
    page_content=('때까지 회사는 보험금지급지연에 따른 이자를 지급하지 않\n'
 '습니다.\n'
 '\uf000 회사는 제6항의 서면조사에 대한 동의 요청시 조사목적,\n'
 '사용처 등을 명시하고 설명합니다.# 제9조(적립부분 적립이율에 관한 사항)\uf000 이 보험의 적립부분 순보험료(적립보험료에서 '
 '계약체결\n'
 '비용 및 계약관리비용을 공제한 금액을 말합니다. 이하 같\n'
 '습니다)에 대한 적립이율은 [보장]공시이율로 합니다.\n'
 '\uf000 [보장]공시이율은 매월 마지막 날(다만, 마지막 날이 영\n'
 '업일이 아닌 때에는 직전의 영업일로 함) 이전에 산출하며,\n'
 '그 다음달에 한하여 적용합니다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
