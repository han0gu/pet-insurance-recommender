from langchain_core.documents import Document

chunk = Document(
    page_content=('- 한 해당 갱신계약의 보험료를 돌려 드립니다.\n'
 '# 제 7조 (준용규정)이 특별약관에 정하지 않은 사항은 보통약관 및 해당 갱신계약을 따릅니다.- 98 -98 / 181# 4-2. '
 '보험료 자동납입 특별약관# 제 1조 (보험료납입)- ① 보험계약자(이하「계약자」라 합니다)는 제2회 이후의 보험료부터 이 특별약관에 따\n'
 '- 라 계약자의 지정계좌를 이용하여 보험료를 자동납입하거나 급여이체를 통하여 납입\n'
 '- 합니다.\n'
 '- ② 제1회 보험료의 납입방법을 계약자의 지정 금융기관 지정계좌를 통한 자동납입으로'),
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
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000541',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
