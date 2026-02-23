from langchain_core.documents import Document

chunk = Document(
    page_content=('- 는 제26조(계약의 해지) 제3항이 적용됩니다.\n'
 '# 제25조[강제집행 등으로 인한 해지계약의 특별부활(효력회복)]① 타인을 위한 계약의 경우 제30조(보험료의 환급)에 따른 계약자의 '
 '환급금 청구권에 대한 강제집행,\n'
 '담보권실행, 국세 및 지방세 체납처분절차에 의해 계약이 해지된 경우에는, 회사는 해지 당시의 피\n'
 '보험자가 계약자의 동의를 얻어 계약 해지로 회사가 채권자에게 지급한 금액을 회사에게 지급하고\n'
 '제19조(계약내용의 변경 등) 제1항의 절차에 따라 계약자 명의를 피보험자로 변경하여 계약의 특'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000063',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
