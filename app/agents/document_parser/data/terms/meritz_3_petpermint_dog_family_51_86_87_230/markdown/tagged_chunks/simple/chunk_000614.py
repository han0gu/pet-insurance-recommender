from langchain_core.documents import Document

chunk = Document(
    page_content=('데 앞니(중절치)간 거리를 기준으로 한다. 단, 가\n'
 '운데 앞니(중절치)가 없는 경우에는 측정가능한 인\n'
 '접 치아간 거리의 최대치를 기준으로 한다.- 6) 부정교합은 위턱(상악)과 아래턱(하악)의 부조화로\n'
 '- 윗니(상악치아)와 아랫니(하악치아)가 전방 및 측\n'
 '- 방으로 맞물림에 제한이 있는 상태를 말한다.\n'
 '- 7) “말하는 기능에 심한 장해를 남긴 때”라 함은 아래\n'
 '- 의 경우 중 하나 이상에 해당되는 때를 말한다.\n'
 '- 가) 언어평가상 자음정확도가 30%미만인 경우\n'
 '- 나) 전실어증, 운동성실어증(브로카실어증)으로 의사'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000614',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
