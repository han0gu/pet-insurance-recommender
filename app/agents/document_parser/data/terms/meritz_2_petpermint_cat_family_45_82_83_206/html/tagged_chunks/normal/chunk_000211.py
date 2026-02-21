from langchain_core.documents import Document

chunk = Document(
    page_content=('. 국세 및 지방세 체납처분 절차란 국세 또는 지방세를<br>체납할 경우 국세 기본법 및 지방세법에 의하여 체납<br>된 세금에 대하여 '
 '가산금 징수, 독촉장 발부 및 재산<br>압류 등의 집행을 하는 것을 말합니다.<br>법원은 채권자의 신청에 따른 강제집행 및 '
 '담보권실행으<br>로 채무자의 해약환급금을 압류할 수 있으며, 법원의 추<br>심명령 또는 전부명령에 따라 회사는 채권자에게 '
 '해약환<br>급금을 지급하게 됩니다.<br>또한, 국세 및 지방세 체납시 국세청 및 지방자치단체에<br>의해 채무자의 해약환급금이 압류될 '
 '수 있으며,'),
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
 'indexing': {'chunk_id': 'chunk_000211',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
