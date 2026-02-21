from langchain_core.documents import Document

chunk = Document(
    page_content=('에 회사는 보험료 납입을 면제 또는 지원하지 않습니다. 다만, 질병으로 인한 사망 또\n'
 '는 진단확정된 질병으로 장해분류표에서 정한 장해지급률이 80% 이상에 해당하는 장\n'
 '해상태가 되어 보험금 지급사유, 보험료 납입면제사유 또는 유사암 납입지원 사유가\n'
 '발생한 경우에는 이를 적용하지 않습니다.1. 【붙임1】(특정신체부위 분류표) 중에서 회사가 지정한 부위(이하「특정신체부위」\n'
 '라 합니다)에 발생한 질병 또는 특정신체부위에 발생한 질병의 전이로 인하여 특정\n'
 '신체부위 이외의 부위에 발생한 질병(단, 전이는 합병증으로 보지 않습니다)'),
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
 'indexing': {'chunk_id': 'chunk_000558',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
