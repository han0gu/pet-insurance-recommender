from langchain_core.documents import Document

chunk = Document(
    page_content=('2(장해분류표)】에 장해판정시기를 별도로 정한 경우에는\n'
 '그에 따릅니다.\n'
 '\uf000 제1항에 따라 장해지급률이 결정되었으나 그 이후 보장\n'
 '을 받을 수 있는 기간(보장의 효력이 없어진 경우에는 보험\n'
 '기간이 10년 이상인 보장은 상해 발생일부터 2년 이내로 하\n'
 '고, 보험기간이 10년 미만인 보장은 상해 발생일부터 1년이\n'
 '내)에 장해상태가 더 악화된 때에는 그 악화된 장해상태를\n'
 '기준으로 장해지급률을 결정합니다.\n'
 '\uf000【별표2(장해분류표)】에 해당되지 않는 후유장해는 피보\n'
 '험자의 직업, 연령, 신분 또는 성별 등에 관계없이 신체의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
