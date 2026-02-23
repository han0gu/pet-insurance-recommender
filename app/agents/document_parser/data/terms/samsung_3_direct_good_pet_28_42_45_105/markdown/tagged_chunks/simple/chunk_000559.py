from langchain_core.documents import Document

chunk = Document(
    page_content=('신체부위 이외의 부위에 발생한 질병(단, 전이는 합병증으로 보지 않습니다)\n'
 '2. 【붙임2】(특정질병 분류표) 중에서 회사가 지정한 질병(이하「특정질병」이라 합\n'
 '니다)- \n'
 '# <용어풀이># [장해지급률]질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.② 제1항의 회사가 보험금을 지급하지 않는 기간(이하 「부담보 기간」이라 합니다)은 특\n'
 '정신체부위 또는 특정질병의 상태에 따라 「1개월부터 5년」또는「보험계약의 보험기'),
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
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000559',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
