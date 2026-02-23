from langchain_core.documents import Document

chunk = Document(
    page_content=('- 따라 법률상의 권리를 행사할 수 있습니다.\n'
 '# <용어풀이># [제척기간]권리관계를 빨리 확정하기 위하여 어떤 종류의 권리에 대하여 법률이 정하고 있는 존속 기간을 말\n'
 '하며, 이 기간이 지나면 해당 권리는 소멸됩니다.# 제31조 (중대사유로 인한 해지)① 회사는 아래와 같은 사실이 있을 경우에는 그 '
 '사실을 안 날부터 1개월 이내에 계약을\n'
 '해지할 수 있습니다.- 1. 계약자, 피보험자 또는 보험수익자가 보험금을 지급받을 목적으로 고의로 보험금\n'
 '- 지급사유를 발생시킨 경우'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000108',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
